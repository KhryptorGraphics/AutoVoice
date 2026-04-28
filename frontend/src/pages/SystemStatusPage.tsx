import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Activity, AlertTriangle, BarChart3, CheckCircle2, Package, RefreshCw, SlidersHorizontal, XCircle, Zap } from 'lucide-react'
import clsx from 'clsx'

import { AudioDeviceSelector } from '../components/AudioDeviceSelector'
import { GPUMonitor } from '../components/GPUMonitor'
import { ModelManager } from '../components/ModelManager'
import { NotificationSettings } from '../components/NotificationSettings'
import { StatusBanner } from '../components/StatusBanner'
import { SystemConfigPanel } from '../components/SystemConfigPanel'
import { TensorRTControls } from '../components/TensorRTControls'
import { useToastContext } from '../contexts/ToastContext'
import { apiService, type AppSettings, type BenchmarkDashboard, type LivePipelineType, type OfflinePipelineType, type ReleaseEvidence } from '../services/api'

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
        benchmarkDashboard,
        releaseEvidence,
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
        apiService.getLatestBenchmarkDashboard().catch(() => null),
        apiService.getLatestReleaseEvidence().catch(() => null),
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
        benchmarkDashboard,
        releaseEvidence,
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

          <BenchmarkEvidencePanel
            dashboard={data.benchmarkDashboard}
            releaseEvidence={data.releaseEvidence}
          />

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

function BenchmarkEvidencePanel({
  dashboard,
  releaseEvidence,
}: {
  dashboard: BenchmarkDashboard | null
  releaseEvidence: ReleaseEvidence | null
}) {
  const pipelines = Object.entries(dashboard?.pipelines ?? {})
  const comparisons = dashboard?.comparisons ?? {}
  const generatedAt = releaseEvidence?.generated_at ?? dashboard?.generated_at
  const freshness = generatedAt ? evidenceFreshness(generatedAt) : null
  const releaseReady = releaseEvidence?.ready_for_release ?? releaseEvidence?.ready
  const qualityPassed = releaseEvidence?.quality_gate_passed ?? Boolean(releaseReady)
  const laneResults = releaseEvidence?.lane_results ?? []
  const failedLanes = laneResults.filter((lane) => lane.status === 'failed' || lane.ok === false)
  const skippedLanes = laneResults.filter((lane) => lane.status === 'skipped' || lane.details?.skipped === true)
  const blockers = releaseEvidence?.blockers ?? []
  const preflightChecks = releaseEvidence?.preflight_checks ?? []
  const failedPreflightChecks = preflightChecks.filter((check) => check.ok === false && check.skipped !== true)
  const artifactEntries = [
    ...Object.entries(releaseEvidence?.artifacts ?? {}),
    ['benchmark_dashboard', 'reports/benchmarks/latest/benchmark_dashboard.json'],
    ['benchmark_release_evidence', 'reports/benchmarks/latest/release_evidence.json'],
    ...(dashboard?.provenance?.source_bundles ?? []).map((bundle, index) => [`source_bundle_${index + 1}`, bundle] as [string, string]),
  ].filter(([, value], index, entries) => value && entries.findIndex(([, candidate]) => candidate === value) === index)
  const nextActions = buildEvidenceActions({
    hasDashboard: Boolean(dashboard),
    hasReleaseEvidence: Boolean(releaseEvidence),
    blockers,
    failedLanes,
    skippedLanes,
    failedPreflightChecks,
    qualityPassed,
  })
  const statusLabel = releaseEvidence?.status
    ?? (releaseReady === true ? 'go' : releaseReady === false ? 'blocked' : qualityPassed ? 'quality passed' : 'needs evidence')
  const hardwareReady = failedPreflightChecks.length === 0 && (releaseReady === true || laneResults.some((lane) => lane.ok === true))

  return (
    <section className="rounded-xl border border-gray-800 bg-gray-900/80 p-6 shadow-lg" data-testid="benchmark-evidence-panel">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <h2 className="flex items-center gap-2 text-lg font-semibold text-white">
            <BarChart3 className="h-5 w-5 text-cyan-300" />
            Benchmark and Release Evidence
          </h2>
          <p className="mt-1 text-sm text-gray-400">
            Current quality gate, fixture tier, and candidate-promotion evidence generated by the benchmark platform.
          </p>
        </div>
        <span className={clsx(
          'inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium',
          qualityPassed ? 'bg-emerald-500/15 text-emerald-200' : 'bg-amber-500/15 text-amber-200'
        )}>
          {qualityPassed ? <CheckCircle2 className="h-3.5 w-3.5" /> : <XCircle className="h-3.5 w-3.5" />}
          {qualityPassed && releaseReady !== false ? 'Evidence ready' : 'Evidence missing or blocked'}
        </span>
      </div>

      {!dashboard || !releaseEvidence ? (
        <StatusBanner
          tone="warning"
          title="Benchmark evidence unavailable"
          message="Run scripts/build_benchmark_dashboard.py and validate release evidence before promoting experimental features."
          compact
        />
      ) : (
        <>
          <div className="mt-5 grid grid-cols-1 gap-4 md:grid-cols-4">
            <DetailCard label="Generated" value={generatedAt ? new Date(generatedAt).toLocaleString() : 'Unknown'} detail={freshness ?? 'No timestamp'} />
            <DetailCard label="Release status" value={statusLabel} detail={releaseEvidence.git_sha_short ? `sha ${releaseEvidence.git_sha_short}` : 'No git SHA'} />
            <DetailCard label="Hardware" value={hardwareReady ? 'Ready' : 'Needs validation'} detail={releaseEvidence.target_hardware ?? dashboard.target_hardware ?? 'Unknown hardware'} />
            <DetailCard label="Lane issues" value={`${failedLanes.length} failed / ${skippedLanes.length} skipped`} detail={`${laneResults.length} lanes reported`} />
          </div>

          {(blockers.length > 0 || failedPreflightChecks.length > 0) && (
            <div className="mt-5 rounded-lg border border-amber-500/30 bg-amber-500/10 p-4" data-testid="evidence-blockers">
              <div className="flex items-center gap-2 text-sm font-semibold text-amber-100">
                <AlertTriangle className="h-4 w-4" />
                Blocking evidence
              </div>
              <ul className="mt-3 space-y-2 text-sm text-amber-100/90">
                {[...blockers.map(formatEvidenceItem), ...failedPreflightChecks.map((check) => `${check.name ?? 'preflight'} failed${check.stderr ? `: ${check.stderr}` : ''}`)].slice(0, 6).map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="mt-5 grid grid-cols-1 gap-4 xl:grid-cols-2">
            <EvidenceList
              title="Failed and skipped lanes"
              empty="No failed or skipped lanes reported."
              items={[...failedLanes, ...skippedLanes].map((lane) => ({
                key: lane.name ?? lane.lane ?? 'lane',
                primary: lane.name ?? lane.lane ?? 'unnamed lane',
                secondary: formatEvidenceItem(lane.details?.reason ?? lane.details?.action ?? lane.stderr ?? lane.status ?? (lane.ok === false ? 'failed' : 'skipped')),
              }))}
              testId="evidence-lane-list"
            />
            <EvidenceList
              title="Artifact references"
              empty="No artifact references reported."
              items={artifactEntries.slice(0, 8).map(([name, value]) => ({
                key: `${name}:${value}`,
                primary: name,
                secondary: value,
              }))}
              testId="evidence-artifact-list"
            />
          </div>

          <EvidenceList
            title="Next actions"
            empty="No follow-up action required by the current evidence."
            items={nextActions.map((action) => ({ key: action, primary: action }))}
            testId="evidence-next-actions"
          />

          <div className="mt-5 overflow-hidden rounded-lg border border-gray-800">
            <div className="grid grid-cols-5 bg-gray-950 px-4 py-2 text-xs font-medium uppercase tracking-wide text-gray-500">
              <span>Pipeline</span>
              <span>Tier</span>
              <span>Samples</span>
              <span>Similarity</span>
              <span>Promotion</span>
            </div>
            {pipelines.map(([name, pipeline]) => {
              const comparison = comparisons[name]
              return (
                <div key={name} className="grid grid-cols-5 border-t border-gray-800 px-4 py-3 text-sm text-gray-300">
                  <span className="font-medium text-white">{name}</span>
                  <span>{pipeline.fixture_tier ?? 'unspecified'}</span>
                  <span>{pipeline.sample_count}</span>
                  <span>{formatMetric(pipeline.summary?.speaker_similarity_mean?.value)}</span>
                  <span className={comparison?.meets_or_beats_canonical ? 'text-emerald-300' : 'text-gray-500'}>
                    {comparison ? (comparison.meets_or_beats_canonical ? 'candidate wins' : `below ${comparison.canonical_pipeline}`) : 'canonical'}
                  </span>
                </div>
              )
            })}
          </div>
        </>
      )}
    </section>
  )
}

function buildEvidenceActions({
  hasDashboard,
  hasReleaseEvidence,
  blockers,
  failedLanes,
  skippedLanes,
  failedPreflightChecks,
  qualityPassed,
}: {
  hasDashboard: boolean
  hasReleaseEvidence: boolean
  blockers: Array<string | Record<string, unknown>>
  failedLanes: NonNullable<ReleaseEvidence['lane_results']>
  skippedLanes: NonNullable<ReleaseEvidence['lane_results']>
  failedPreflightChecks: NonNullable<ReleaseEvidence['preflight_checks']>
  qualityPassed: boolean
}): string[] {
  if (!hasDashboard || !hasReleaseEvidence) {
    return ['Generate benchmark dashboard and release evidence before promotion.']
  }
  if (blockers.length > 0) {
    return blockers.map(formatEvidenceItem)
  }
  if (failedPreflightChecks.length > 0) {
    return failedPreflightChecks.map((check) => `Fix ${check.name ?? 'preflight'} and rerun hardware release evidence.`)
  }
  if (failedLanes.length > 0) {
    return failedLanes.map((lane) => `Rerun or fix ${lane.name ?? lane.lane ?? 'failed lane'}.`)
  }
  if (skippedLanes.length > 0) {
    return skippedLanes.map((lane) => formatEvidenceItem(lane.details?.action ?? `Run ${lane.name ?? lane.lane ?? 'skipped lane'} with required services or hardware.`))
  }
  if (!qualityPassed) {
    return ['Investigate quality gate failures and rebuild benchmark evidence.']
  }
  return ['Evidence is ready for operator review.']
}

function EvidenceList({
  title,
  empty,
  items,
  testId,
}: {
  title: string
  empty: string
  items: Array<{ key: string; primary: string; secondary?: string }>
  testId: string
}) {
  return (
    <div className="mt-5 rounded-lg border border-gray-800 bg-gray-950/70 p-4" data-testid={testId}>
      <h3 className="text-sm font-semibold text-white">{title}</h3>
      {items.length === 0 ? (
        <p className="mt-2 text-sm text-gray-500">{empty}</p>
      ) : (
        <div className="mt-3 space-y-3">
          {items.map((item) => (
            <div key={item.key} className="rounded-md border border-gray-800 bg-gray-900/70 p-3">
              <div className="break-all text-sm font-medium text-gray-100">{item.primary}</div>
              {item.secondary && <div className="mt-1 break-all text-xs text-gray-400">{item.secondary}</div>}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function formatEvidenceItem(item: unknown): string {
  if (typeof item === 'string') return item
  if (item && typeof item === 'object') {
    const record = item as Record<string, unknown>
    return String(record.details ?? record.reason ?? record.action ?? record.check ?? JSON.stringify(record))
  }
  return String(item ?? 'unknown')
}

function DetailCard({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-950/70 p-4">
      <div className="text-xs uppercase tracking-wide text-gray-500">{label}</div>
      <div className="mt-2 truncate text-lg font-semibold text-white">{value}</div>
      <div className="mt-1 truncate text-xs text-gray-400">{detail}</div>
    </div>
  )
}

function evidenceFreshness(generatedAt: string): string {
  const ageMs = Date.now() - new Date(generatedAt).getTime()
  if (!Number.isFinite(ageMs)) return 'Invalid timestamp'
  const ageHours = Math.max(0, ageMs / 3_600_000)
  if (ageHours < 1) return 'Generated less than 1h ago'
  if (ageHours < 48) return `Generated ${ageHours.toFixed(0)}h ago`
  return `Generated ${(ageHours / 24).toFixed(1)}d ago`
}

function formatMetric(value?: number): string {
  return typeof value === 'number' ? value.toFixed(3) : 'n/a'
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
