import { useQuery } from '@tanstack/react-query'
import { Activity, Package, Zap, CheckCircle, XCircle } from 'lucide-react'
import clsx from 'clsx'
import { apiService } from '../services/api'
import { GPUMonitor } from '../components/GPUMonitor'

export function SystemStatusPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['systemDashboard'],
    queryFn: async () => {
      const [
        systemStatus,
        modelsInfo,
        healthStatus,
        appSettings,
        deviceConfig,
        audioRouterConfig,
        pitchConfig,
        separationConfig,
        pipelineStatus,
      ] = await Promise.all([
        apiService.getSystemStatus(),
        apiService.getModelsInfo(),
        apiService.healthCheck(),
        apiService.getAppSettings(),
        apiService.getDeviceConfig(),
        apiService.getAudioRouterConfig(),
        apiService.getPitchConfig(),
        apiService.getSeparationConfig(),
        apiService.getPipelineStatus(),
      ])

      return {
        systemStatus,
        modelsInfo,
        healthStatus,
        appSettings,
        deviceConfig,
        audioRouterConfig,
        pitchConfig,
        separationConfig,
        pipelineStatus,
      }
    },
    refetchInterval: 5000,
  })

  const modelsInfo = data?.modelsInfo
  const healthStatus = data?.healthStatus
  const pipelineEntries = Object.entries(data?.pipelineStatus?.pipelines ?? {})
  const loadedPipelineCount = pipelineEntries.filter(([, status]) => status.loaded).length

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 flex items-center space-x-3">
          <Activity className="w-8 h-8 text-purple-600" />
          <span>System Status</span>
        </h1>
        <p className="text-gray-600 mt-2">
          Monitor GPU utilization, loaded models, and live runtime configuration.
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
          <p className="text-red-700 text-sm mt-1">{(error as Error).message}</p>
        </div>
      )}

      {data && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <GPUMonitor refreshInterval={2000} />
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center space-x-2">
              <Zap className="w-5 h-5 text-purple-600" />
              <span>Health Status</span>
            </h3>

            {healthStatus && (
              <div className="space-y-3">
                <StatusItem label="API Server" status={healthStatus.status === 'healthy' ? 'online' : healthStatus.status} />
                <StatusItem label="GPU" status={healthStatus.gpu_available ? 'available' : 'unavailable'} />
                <StatusItem label="Loaded Models" status={healthStatus.models_loaded ? 'loaded' : 'idle'} />
                <StatusItem label="Loaded Pipelines" status={loadedPipelineCount > 0 ? 'loaded' : 'idle'} />

                <div className="pt-3 border-t border-gray-200 space-y-2 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Uptime:</span>
                    <span className="font-medium text-gray-900">{formatUptime(healthStatus.uptime)}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Platform:</span>
                    <span className="font-medium text-gray-900">{data.systemStatus.device}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Python:</span>
                    <span className="font-medium text-gray-900">{data.systemStatus.python_version.split(' ')[0]}</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="lg:col-span-3 bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center space-x-2">
              <Package className="w-5 h-5 text-purple-600" />
              <span>Loaded Models</span>
            </h3>

            {modelsInfo?.models?.length ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {modelsInfo.models.map((model, index) => (
                  <div key={`${model.model_type}-${index}`} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex items-start justify-between mb-2">
                      <h4 className="font-semibold text-gray-900">{model.name}</h4>
                      {model.loaded ? (
                        <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0" />
                      ) : (
                        <XCircle className="w-5 h-5 text-gray-400 flex-shrink-0" />
                      )}
                    </div>
                    <div className="space-y-1 text-sm">
                      <DetailRow label="Type" value={model.model_type} />
                      <DetailRow label="Backend" value={model.runtime_backend ?? 'unknown'} />
                      <DetailRow label="Device" value={model.device ?? 'n/a'} />
                      <DetailRow label="Memory" value={formatBytes(model.memory_usage ?? 0)} />
                      <DetailRow label="Source" value={model.source ?? 'registry'} />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-gray-600">No models are currently loaded.</p>
            )}
          </div>

          <div className="lg:col-span-3 bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-bold text-gray-900 mb-4">Runtime Configuration</h3>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <section>
                <h4 className="font-semibold text-gray-700 mb-3">Audio</h4>
                <div className="space-y-2 text-sm">
                  <DetailRow label="Device Sample Rate" value={`${data.deviceConfig.sample_rate} Hz`} />
                  <DetailRow label="Router Sample Rate" value={`${data.audioRouterConfig.sample_rate} Hz`} />
                  <DetailRow label="Input Device" value={data.deviceConfig.input_device_id ?? 'system default'} />
                  <DetailRow label="Output Device" value={data.deviceConfig.output_device_id ?? 'system default'} />
                </div>
              </section>

              <section>
                <h4 className="font-semibold text-gray-700 mb-3">Conversion Defaults</h4>
                <div className="space-y-2 text-sm">
                  <DetailRow label="Offline Pipeline" value={data.appSettings.preferred_offline_pipeline} />
                  <DetailRow label="Live Pipeline" value={data.appSettings.preferred_live_pipeline} />
                  <DetailRow label="Pitch Method" value={data.pitchConfig.method} />
                  <DetailRow label="Separator Model" value={data.separationConfig.model} />
                </div>
              </section>

              <section>
                <h4 className="font-semibold text-gray-700 mb-3">Routing & Performance</h4>
                <div className="space-y-2 text-sm">
                  <DetailRow label="Speaker Output" value={data.audioRouterConfig.speaker_enabled ? 'enabled' : 'disabled'} />
                  <DetailRow label="Headphone Output" value={data.audioRouterConfig.headphone_enabled ? 'enabled' : 'disabled'} />
                  <DetailRow label="GPU Acceleration" value={data.systemStatus.cuda_available ? 'enabled' : 'disabled'} />
                  <DetailRow label="Loaded Pipelines" value={String(loadedPipelineCount)} />
                </div>
              </section>
            </div>
          </div>

          <div className="lg:col-span-3 bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-bold text-gray-900 mb-4">Pipeline Matrix</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
              {pipelineEntries.map(([pipeline, status]) => (
                <div key={pipeline} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold text-gray-900">{pipeline}</span>
                    <span className={clsx('text-xs font-medium px-2 py-1 rounded-full', status.loaded ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600')}>
                      {status.loaded ? 'loaded' : 'idle'}
                    </span>
                  </div>
                  <div className="space-y-1 text-sm">
                    <DetailRow label="Sample Rate" value={status.sample_rate ? `${status.sample_rate} Hz` : 'n/a'} />
                    <DetailRow label="Latency Target" value={status.latency_target_ms ? `${status.latency_target_ms} ms` : 'n/a'} />
                    <DetailRow label="Memory" value={status.memory_gb ? `${status.memory_gb.toFixed(2)} GB` : '0 GB'} />
                  </div>
                  {status.description && (
                    <p className="text-xs text-gray-500 mt-3">{status.description}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function StatusItem({ label, status }: { label: string; status: string }) {
  const normalized = status.toLowerCase()
  const isOnline = ['online', 'available', 'loaded', 'connected', 'healthy'].includes(normalized)

  return (
    <div className="flex items-center justify-between">
      <span className="text-sm text-gray-600">{label}:</span>
      <div className="flex items-center space-x-2">
        <div className={clsx('w-2 h-2 rounded-full', isOnline ? 'bg-green-500' : 'bg-gray-400')} />
        <span className={clsx('text-sm font-medium', isOnline ? 'text-green-600' : 'text-gray-500')}>
          {status}
        </span>
      </div>
    </div>
  )
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4">
      <span className="text-gray-600">{label}:</span>
      <span className="font-medium text-gray-900 text-right">{value}</span>
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
  if (!bytes) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(k)), sizes.length - 1)
  return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`
}
