import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { AlertTriangle, CheckCircle2, Filter, ListChecks, XCircle } from 'lucide-react'
import clsx from 'clsx'

import { StatusBanner } from '../components/StatusBanner'
import { apiService, type TrainingSample } from '../services/api'

type SampleFilter = 'all' | 'trainable' | 'blocked'

function formatSeconds(value?: number | null): string {
  if (!Number.isFinite(value ?? NaN)) return 'n/a'
  const seconds = Math.max(0, Number(value))
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`
}

function formatRatio(value?: number | null): string {
  return typeof value === 'number' ? `${Math.round(value * 100)}%` : 'n/a'
}

function statusTone(sample: TrainingSample) {
  if (sample.trainable) return 'text-emerald-300 bg-emerald-500/10 border-emerald-500/30'
  if (sample.quality_status === 'fail') return 'text-red-300 bg-red-500/10 border-red-500/30'
  return 'text-amber-300 bg-amber-500/10 border-amber-500/30'
}

function sampleSourceLabel(sample: TrainingSample): string {
  const sourceFile = sample.metadata?.source_file
  if (sample.source) return sample.source
  if (typeof sourceFile === 'string') return sourceFile
  return sample.id
}

export function SampleInboxPage() {
  const [filter, setFilter] = useState<SampleFilter>('all')
  const query = useQuery({
    queryKey: ['sampleReview', filter],
    queryFn: () => apiService.listSampleReview({
      trainable: filter === 'all' ? undefined : filter === 'trainable',
    }),
    refetchInterval: 10000,
  })

  const samples = useMemo(() => query.data?.samples ?? [], [query.data?.samples])
  const totalDuration = useMemo(
    () => samples.reduce((sum, sample) => sum + Number(sample.duration_seconds || 0), 0),
    [samples],
  )

  return (
    <div className="mx-auto max-w-7xl space-y-6 px-4 py-8" data-testid="sample-inbox-page">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h1 className="flex items-center gap-3 text-3xl font-bold text-white">
            <ListChecks className="h-8 w-8 text-emerald-300" />
            Training Sample Inbox
          </h1>
          <p className="mt-2 max-w-3xl text-gray-400">
            Review every local training sample before retraining, including source, consent, loudness, silence, clipping, and remediation.
          </p>
        </div>
        <div className="flex items-center gap-2 rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-300">
          <Filter className="h-4 w-4" />
          <select
            value={filter}
            onChange={(event) => setFilter(event.target.value as SampleFilter)}
            className="bg-transparent text-sm text-white outline-none"
            data-testid="sample-inbox-filter"
          >
            <option className="bg-gray-900" value="all">All samples</option>
            <option className="bg-gray-900" value="trainable">Trainable</option>
            <option className="bg-gray-900" value="blocked">Blocked</option>
          </select>
        </div>
      </div>

      {query.error && (
        <StatusBanner
          tone="danger"
          title="Sample inbox unavailable"
          message={(query.error as Error).message}
        />
      )}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <Summary label="Samples" value={String(query.data?.count ?? samples.length)} />
        <Summary label="Trainable" value={String(query.data?.summary.trainable ?? 0)} tone="success" />
        <Summary label="Blocked" value={String(query.data?.summary.blocked ?? 0)} tone="warning" />
        <Summary label="Duration" value={formatSeconds(totalDuration)} />
      </div>

      <section className="overflow-hidden rounded-xl border border-gray-800 bg-gray-900/80">
        <div className="grid grid-cols-[1.2fr_0.9fr_0.7fr_0.7fr_0.7fr_1fr] gap-3 bg-gray-950 px-4 py-3 text-xs font-medium uppercase tracking-wide text-gray-500">
          <span>Profile / Source</span>
          <span>Quality</span>
          <span>Duration</span>
          <span>Loudness</span>
          <span>Silence / Clip</span>
          <span>Remediation</span>
        </div>
        {query.isLoading ? (
          <div className="px-4 py-10 text-center text-gray-500">Loading samples...</div>
        ) : samples.length === 0 ? (
          <div className="px-4 py-10 text-center text-gray-500">No samples match this filter.</div>
        ) : (
          samples.map((sample) => (
            <div
              key={`${sample.profile_id}-${sample.id}`}
              className="grid grid-cols-[1.2fr_0.9fr_0.7fr_0.7fr_0.7fr_1fr] gap-3 border-t border-gray-800 px-4 py-4 text-sm text-gray-300"
            >
              <div className="min-w-0">
                <div className="truncate font-medium text-white">{sample.profile_name ?? sample.profile_id}</div>
                <div className="mt-1 truncate text-xs text-gray-500">{sampleSourceLabel(sample)}</div>
                <div className="mt-1 text-xs text-gray-500">{sample.consent_status ?? 'unknown consent'}</div>
              </div>
              <div>
                <span className={clsx('inline-flex items-center gap-1 rounded-full border px-2 py-1 text-xs', statusTone(sample))}>
                  {sample.trainable ? <CheckCircle2 className="h-3 w-3" /> : sample.quality_status === 'fail' ? <XCircle className="h-3 w-3" /> : <AlertTriangle className="h-3 w-3" />}
                  {sample.trainable ? 'trainable' : 'blocked'} · {sample.quality_status ?? 'unknown'}
                </span>
                {sample.issues && sample.issues.length > 0 && (
                  <div className="mt-2 text-xs text-gray-500">{sample.issues.join(', ')}</div>
                )}
              </div>
              <div>{formatSeconds(sample.duration_seconds)}</div>
              <div>{typeof sample.rms_loudness === 'number' ? sample.rms_loudness.toFixed(4) : 'n/a'}</div>
              <div>{formatRatio(sample.silence_ratio)} / {formatRatio(sample.clipping_ratio)}</div>
              <div className="text-xs text-gray-400">
                {sample.recommendations?.[0] ?? 'No remediation needed.'}
              </div>
            </div>
          ))
        )}
      </section>
    </div>
  )
}

function Summary({ label, value, tone = 'info' }: { label: string; value: string; tone?: 'info' | 'success' | 'warning' }) {
  const toneClass = tone === 'success'
    ? 'border-emerald-500/20 bg-emerald-500/10'
    : tone === 'warning'
      ? 'border-amber-500/20 bg-amber-500/10'
      : 'border-cyan-500/20 bg-cyan-500/10'
  return (
    <div className={clsx('rounded-xl border p-4', toneClass)}>
      <div className="text-sm text-gray-400">{label}</div>
      <div className="mt-2 text-2xl font-semibold text-white">{value}</div>
    </div>
  )
}
