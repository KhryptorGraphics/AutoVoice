import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Activity, Package, RefreshCw, SlidersHorizontal, Zap } from 'lucide-react'
import clsx from 'clsx'

import { AudioDeviceSelector } from '../components/AudioDeviceSelector'
import { GPUMonitor } from '../components/GPUMonitor'
import { ModelManager } from '../components/ModelManager'
import { NotificationSettings } from '../components/NotificationSettings'
import { StatusBanner } from '../components/StatusBanner'
import { SystemConfigPanel } from '../components/SystemConfigPanel'
import { TensorRTControls } from '../components/TensorRTControls'
import { useToastContext } from '../contexts/ToastContext'
import { apiService, type AppSettings, type LivePipelineType, type OfflinePipelineType } from '../services/api'

const OFFLINE_PIPELINES: Array<{ value: OfflinePipelineType; label: string }> = [
  { value: 'quality_seedvc', label: 'quality_seedvc' },
  { value: 'quality_shortcut', label: 'quality_shortcut' },
  { value: 'quality', label: 'quality' },
  { value: 'realtime', label: 'realtime' },
]

const LIVE_PIPELINES: Array<{ value: LivePipelineType; label: string }> = [
  { value: 'realtime', label: 'realtime' },
  { value: 'realtime_meanvc', label: 'realtime_meanvc' },
]

export function SystemStatusPage() {
  const queryClient = useQueryClient()
  const toast = useToastContext()
  const [runtimePrefs, setRuntimePrefs] = useState<Pick<AppSettings, 'preferred_offline_pipeline' | 'preferred_live_pipeline'>>({
    preferred_offline_pipeline: 'quality_seedvc',
    preferred_live_pipeline: 'realtime',
  })

  const { data, isLoading, error, refetch, isFetching } = useQuery({
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

  useEffect(() => {
    if (!data?.appSettings) {
      return
    }
    setRuntimePrefs({
      preferred_offline_pipeline: data.appSettings.preferred_offline_pipeline,
      preferred_live_pipeline: data.appSettings.preferred_live_pipeline,
    })
  }, [data?.appSettings])

  const updateRuntimeMutation = useMutation({
    mutationFn: (nextSettings: Partial<AppSettings>) => apiService.updateAppSettings(nextSettings),
    onSuccess: (nextSettings) => {
      queryClient.setQueryData(['systemDashboard'], (previous: typeof data) => (
        previous
          ? { ...previous, appSettings: { ...previous.appSettings, ...nextSettings } }
          : previous
      ))
      queryClient.invalidateQueries({ queryKey: ['systemDashboard'] })
      toast.success('Runtime defaults updated')
    },
    onError: (mutationError) => {
      toast.error(mutationError instanceof Error ? mutationError.message : 'Failed to update runtime defaults')
    },
  })

  const modelsInfo = data?.modelsInfo
  const healthStatus = data?.healthStatus
  const pipelineEntries = Object.entries(data?.pipelineStatus?.pipelines ?? {})
  const loadedPipelineCount = pipelineEntries.filter(([, status]) => status.loaded).length
  const loadedModelCount = modelsInfo?.models?.filter((model) => model.loaded).length ?? 0
  const runtimePrefsDirty = useMemo(() => {
    if (!data?.appSettings) {
      return false
    }
    return (
      runtimePrefs.preferred_offline_pipeline !== data.appSettings.preferred_offline_pipeline
      || runtimePrefs.preferred_live_pipeline !== data.appSettings.preferred_live_pipeline
    )
  }, [data?.appSettings, runtimePrefs])

  return (
    <div className="mx-auto max-w-7xl space-y-6 px-4 py-8">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h1 className="flex items-center gap-3 text-3xl font-bold text-white">
            <Activity className="h-8 w-8 text-cyan-400" />
            <span>Operator Console</span>
          </h1>
          <p className="mt-2 max-w-3xl text-gray-400">
            Control model residency, TensorRT builds, routing, and runtime defaults from one place while the backend health
            and GPU telemetry stay visible.
          </p>
        </div>
        <button
          type="button"
          onClick={() => refetch()}
          disabled={isFetching}
          className="inline-flex items-center gap-2 self-start rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200 hover:bg-gray-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <RefreshCw className={clsx('h-4 w-4', isFetching && 'animate-spin')} />
          Refresh console
        </button>
      </div>

      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <div className="h-12 w-12 animate-spin rounded-full border-b-2 border-cyan-500" />
        </div>
      )}

      {error && (
        <StatusBanner
          tone="danger"
          title="Failed to load operator console"
          message={(error as Error).message}
          testId="system-status-error"
        />
      )}

      {data && (
        <>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
            <SummaryCard
              icon={<Activity className="h-5 w-5 text-cyan-300" />}
              label="Backend Health"
              value={healthStatus?.status === 'healthy' ? 'Healthy' : healthStatus?.status ?? 'Unknown'}
              tone={healthStatus?.status === 'healthy' ? 'success' : 'warning'}
              detail={`Uptime ${formatUptime(healthStatus?.uptime ?? 0)}`}
            />
            <SummaryCard
              icon={<Package className="h-5 w-5 text-violet-300" />}
              label="Loaded Models"
              value={String(loadedModelCount)}
              tone={loadedModelCount > 0 ? 'success' : 'warning'}
              detail={`${modelsInfo?.models?.length ?? 0} tracked model slots`}
            />
            <SummaryCard
              icon={<Zap className="h-5 w-5 text-amber-300" />}
              label="Loaded Pipelines"
              value={String(loadedPipelineCount)}
              tone={loadedPipelineCount > 0 ? 'success' : 'warning'}
              detail={`${pipelineEntries.length} runtime profiles`}
            />
            <SummaryCard
              icon={<SlidersHorizontal className="h-5 w-5 text-emerald-300" />}
              label="Runtime Defaults"
              value={`${data.appSettings.preferred_offline_pipeline} / ${data.appSettings.preferred_live_pipeline}`}
              tone="info"
              detail={`${data.pitchConfig.method} pitch · ${data.separationConfig.model} separator`}
            />
          </div>

          {updateRuntimeMutation.error && (
            <StatusBanner
              tone="danger"
              title="Runtime preference update failed"
              message={(updateRuntimeMutation.error as Error).message}
              compact
            />
          )}

          <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
            <div className="xl:col-span-2">
              <GPUMonitor refreshInterval={2000} />
            </div>

            <section className="rounded-xl border border-gray-800 bg-gray-900/80 p-6 shadow-lg">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-white">Runtime Defaults</h2>
                  <p className="mt-1 text-sm text-gray-400">
                    Set the canonical offline and live pipelines used by the rest of the app.
                  </p>
                </div>
                {runtimePrefsDirty && (
                  <span className="rounded-full border border-amber-500/40 bg-amber-500/10 px-3 py-1 text-xs font-medium text-amber-200">
                    Unsaved
                  </span>
                )}
              </div>

              <div className="mt-6 space-y-4">
                <div>
                  <label className="mb-2 block text-sm text-gray-400">Offline default</label>
                  <select
                    data-testid="system-offline-pipeline-select"
                    value={runtimePrefs.preferred_offline_pipeline}
                    onChange={(event) => setRuntimePrefs((prev) => ({
                      ...prev,
                      preferred_offline_pipeline: event.target.value as OfflinePipelineType,
                    }))}
                    className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-white"
                  >
                    {OFFLINE_PIPELINES.map((pipeline) => (
                      <option key={pipeline.value} value={pipeline.value}>
                        {pipeline.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="mb-2 block text-sm text-gray-400">Live default</label>
                  <select
                    data-testid="system-live-pipeline-select"
                    value={runtimePrefs.preferred_live_pipeline}
                    onChange={(event) => setRuntimePrefs((prev) => ({
                      ...prev,
                      preferred_live_pipeline: event.target.value as LivePipelineType,
                    }))}
                    className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-white"
                  >
                    {LIVE_PIPELINES.map((pipeline) => (
                      <option key={pipeline.value} value={pipeline.value}>
                        {pipeline.label}
                      </option>
                    ))}
                  </select>
                </div>

                <button
                  type="button"
                  data-testid="system-runtime-save"
                  disabled={!runtimePrefsDirty || updateRuntimeMutation.isPending}
                  onClick={() => updateRuntimeMutation.mutate(runtimePrefs)}
                  className="inline-flex items-center gap-2 rounded-lg bg-cyan-500 px-4 py-2 text-sm font-medium text-gray-950 hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <SlidersHorizontal className="h-4 w-4" />
                  {updateRuntimeMutation.isPending ? 'Saving defaults...' : 'Save runtime defaults'}
                </button>
              </div>

              <div className="mt-6 grid grid-cols-1 gap-3 border-t border-gray-800 pt-6 text-sm text-gray-300">
                <DetailPanel label="Device sample rate" value={`${data.deviceConfig.sample_rate} Hz`} />
                <DetailPanel
                  label="Audience routing"
                  value={data.audioRouterConfig.speaker_enabled ? `Speaker ${String(data.audioRouterConfig.speaker_device ?? 'default')}` : 'Disabled'}
                />
                <DetailPanel
                  label="Monitor routing"
                  value={data.audioRouterConfig.headphone_enabled ? `Headphone ${String(data.audioRouterConfig.headphone_device ?? 'default')}` : 'Disabled'}
                />
                <DetailPanel label="Platform" value={data.systemStatus.device} />
                <DetailPanel label="Python" value={data.systemStatus.python_version.split(' ')[0]} />
                <DetailPanel label="Separator" value={data.separationConfig.model} />
              </div>
            </section>
          </div>

          <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
            <section className="space-y-6">
              <ModelManager />
              <AudioDeviceSelector />
            </section>
            <section className="space-y-6">
              <TensorRTControls />
              <NotificationSettings />
              <SystemConfigPanel onConfigChange={() => void refetch()} />
            </section>
          </div>

          <section className="rounded-xl border border-gray-800 bg-gray-900/80 p-6 shadow-lg">
            <h2 className="text-lg font-semibold text-white">Pipeline Matrix</h2>
            <p className="mt-1 text-sm text-gray-400">
              Live visibility into which pipelines are actually loaded, how much memory they hold, and their target latency.
            </p>
            <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
              {pipelineEntries.map(([pipeline, status]) => (
                <div key={pipeline} className="rounded-xl border border-gray-800 bg-gray-950/70 p-4">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-white">{pipeline}</span>
                    <span className={clsx(
                      'rounded-full px-2 py-1 text-xs font-medium',
                      status.loaded ? 'bg-emerald-500/15 text-emerald-200' : 'bg-gray-800 text-gray-400'
                    )}>
                      {status.loaded ? 'loaded' : 'idle'}
                    </span>
                  </div>
                  <div className="mt-3 space-y-2 text-sm text-gray-300">
                    <DetailPanel label="Sample rate" value={status.sample_rate ? `${status.sample_rate} Hz` : 'n/a'} />
                    <DetailPanel label="Latency target" value={status.latency_target_ms ? `${status.latency_target_ms} ms` : 'n/a'} />
                    <DetailPanel label="Memory" value={status.memory_gb ? `${status.memory_gb.toFixed(2)} GB` : '0 GB'} />
                  </div>
                  {status.description && (
                    <p className="mt-3 text-xs text-gray-500">{status.description}</p>
                  )}
                </div>
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  )
}

function SummaryCard({
  icon,
  label,
  value,
  detail,
  tone,
}: {
  icon: ReactNode
  label: string
  value: string
  detail: string
  tone: 'info' | 'success' | 'warning'
}) {
  const toneClass = tone === 'success'
    ? 'border-emerald-500/20 bg-emerald-500/10'
    : tone === 'warning'
      ? 'border-amber-500/20 bg-amber-500/10'
      : 'border-cyan-500/20 bg-cyan-500/10'

  return (
    <div className={clsx('rounded-xl border p-4 shadow-lg', toneClass)}>
      <div className="flex items-center justify-between">
        <div className="text-sm text-gray-300">{label}</div>
        {icon}
      </div>
      <div className="mt-3 text-2xl font-semibold text-white">{value}</div>
      <div className="mt-1 text-xs text-gray-400">{detail}</div>
    </div>
  )
}

function DetailPanel({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="text-gray-500">{label}</span>
      <span className="text-right font-medium text-gray-200">{value}</span>
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
