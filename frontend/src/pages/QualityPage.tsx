import { useState, type FormEvent } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Gauge } from 'lucide-react'
import { useSearchParams } from 'react-router-dom'
import clsx from 'clsx'

import { QualityComparisonPanel } from '../components/QualityComparisonPanel'
import { QualityMetricsPanel } from '../components/QualityMetricsPanel'
import { StatusBanner } from '../components/StatusBanner'
import { useToastContext } from '../contexts/ToastContext'
import {
  apiService,
  type ConversionAnalysis,
  type MethodologyComparison,
} from '../services/api'

function asString(value: unknown): string | null {
  return typeof value === 'string' && value.length > 0 ? value : null
}

function formatNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(3)
}

function numericEntries(record: Record<string, unknown>): Array<[string, number]> {
  return Object.entries(record).filter(
    (entry): entry is [string, number] => typeof entry[1] === 'number'
  )
}

export function QualityPage() {
  const toast = useToastContext()
  const [searchParams] = useSearchParams()
  const jobId = searchParams.get('job_id')
  const [selectedProfileId, setSelectedProfileId] = useState<string | null>(null)
  const [checkingDegradation, setCheckingDegradation] = useState(false)

  const { data, isLoading, error } = useQuery({
    queryKey: ['qualityOverview'],
    queryFn: async () => {
      const [profilesQuality, loraAudit] = await Promise.all([
        apiService.getAllProfilesQuality(),
        apiService.auditLoras('summary').catch(() => null),
      ])
      return { profilesQuality, loraAudit }
    },
    refetchInterval: 15000,
  })

  const historyQuery = useQuery({
    queryKey: ['profileQualityHistory', selectedProfileId],
    queryFn: () => apiService.getProfileQualityHistory(selectedProfileId as string),
    enabled: Boolean(selectedProfileId),
  })

  const profiles = data?.profilesQuality.profiles ?? []
  const audit = data?.loraAudit
  const selectedProfile = profiles.find(
    (profile) => asString(profile.profile_id) === selectedProfileId
  )
  const historyEntries = historyQuery.data ?? []
  const historyColumns = Array.from(
    new Set(historyEntries.flatMap((entry) => numericEntries(entry).map(([key]) => key)))
  )

  const checkDegradation = async () => {
    if (!selectedProfileId) return
    setCheckingDegradation(true)
    try {
      const result = await apiService.checkProfileDegradation(selectedProfileId)
      if (result.degraded) {
        toast.error(`Profile degraded${result.reason ? `: ${String(result.reason)}` : ''}`)
      } else {
        toast.success('No degradation detected')
      }
    } catch (checkError) {
      toast.error(checkError instanceof Error ? checkError.message : 'Degradation check failed')
    } finally {
      setCheckingDegradation(false)
    }
  }

  return (
    <div className="mx-auto max-w-7xl space-y-6 px-4 py-8">
      <div>
        <h1 className="flex items-center gap-3 text-3xl font-bold text-white">
          <Gauge className="h-8 w-8 text-cyan-400" />
          <span>Quality</span>
        </h1>
        <p className="mt-2 max-w-3xl text-gray-400">
          Live conversion quality, adapter health, and degradation monitoring across profiles.
        </p>
      </div>

      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <div className="h-12 w-12 animate-spin rounded-full border-b-2 border-cyan-500" />
        </div>
      )}

      {error && (
        <StatusBanner
          tone="danger"
          title="Failed to load quality overview"
          message={(error as Error).message}
          testId="quality-overview-error"
        />
      )}

      {data && (
        <>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-6">
            <StatCard label="Profiles tracked" value={data.profilesQuality.total} />
            <StatCard
              label="Degraded"
              value={data.profilesQuality.degraded_count}
              tone={data.profilesQuality.degraded_count > 0 ? 'warning' : 'default'}
            />
            <StatCard
              label="Critical"
              value={data.profilesQuality.critical_count}
              tone={data.profilesQuality.critical_count > 0 ? 'danger' : 'default'}
            />
            <StatCard label="With adapters" value={audit?.profiles_with_adapters ?? 'n/a'} />
            <StatCard
              label="Needing retrain"
              value={audit?.profiles_needing_retrain ?? 'n/a'}
              tone={(audit?.profiles_needing_retrain ?? 0) > 0 ? 'warning' : 'default'}
            />
            <StatCard
              label="Stale adapters"
              value={audit?.stale_adapters ?? 'n/a'}
              tone={(audit?.stale_adapters ?? 0) > 0 ? 'warning' : 'default'}
            />
          </div>

          <section className="rounded-xl border border-gray-800 bg-gray-900/80 p-6 shadow-lg">
            <h2 className="text-lg font-semibold text-white">Profiles</h2>
            <p className="mt-1 text-sm text-gray-400">
              Click a profile to inspect its quality history and adapter comparison.
            </p>
            {profiles.length === 0 ? (
              <p className="mt-4 text-sm text-gray-500">No profile quality data reported yet.</p>
            ) : (
              <div className="mt-4 overflow-hidden rounded-lg border border-gray-800" data-testid="quality-profiles-table">
                <div className="grid grid-cols-3 bg-gray-950 px-4 py-2 text-xs font-medium uppercase tracking-wide text-gray-500">
                  <span>Profile</span>
                  <span>Status</span>
                  <span>Metrics</span>
                </div>
                {profiles.map((profile) => {
                  const profileId = asString(profile.profile_id) ?? asString(profile.id) ?? ''
                  const name = asString(profile.name) ?? profileId
                  const status = asString(profile.status) ?? 'unknown'
                  const metrics = numericEntries(profile)
                  return (
                    <button
                      key={profileId}
                      type="button"
                      onClick={() => setSelectedProfileId(profileId)}
                      className={clsx(
                        'grid w-full grid-cols-3 border-t border-gray-800 px-4 py-3 text-left text-sm transition-colors hover:bg-gray-800/60',
                        selectedProfileId === profileId && 'bg-gray-800/80'
                      )}
                    >
                      <span>
                        <span className="font-medium text-white">{name}</span>
                        <span className="mt-0.5 block text-xs text-gray-500">{profileId}</span>
                      </span>
                      <span>
                        <span
                          className={clsx(
                            'rounded-full px-2 py-1 text-xs font-medium',
                            status === 'critical'
                              ? 'bg-red-500/15 text-red-300'
                              : status === 'degraded'
                                ? 'bg-amber-500/15 text-amber-200'
                                : 'bg-emerald-500/15 text-emerald-200'
                          )}
                        >
                          {status}
                        </span>
                      </span>
                      <span className="text-gray-300">
                        {metrics.length === 0
                          ? 'n/a'
                          : metrics.map(([key, value]) => `${key} ${formatNumber(value)}`).join(' · ')}
                      </span>
                    </button>
                  )
                })}
              </div>
            )}
          </section>

          {selectedProfileId && (
            <section
              className="space-y-6 rounded-xl border border-gray-800 bg-gray-900/80 p-6 shadow-lg"
              data-testid="quality-profile-detail"
            >
              <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-white">
                    {asString(selectedProfile?.name) ?? selectedProfileId}
                  </h2>
                  <p className="mt-1 text-sm text-gray-400">
                    Quality history and adapter comparison for this profile.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={checkDegradation}
                  disabled={checkingDegradation}
                  className="rounded-lg bg-cyan-600 px-4 py-2 text-sm font-medium text-white hover:bg-cyan-500 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {checkingDegradation ? 'Checking...' : 'Check degradation'}
                </button>
              </div>

              <div>
                <h3 className="text-sm font-semibold text-white">Quality history (30 days)</h3>
                {historyQuery.isLoading ? (
                  <p className="mt-2 text-sm text-gray-500">Loading history...</p>
                ) : historyEntries.length === 0 ? (
                  <p className="mt-2 text-sm text-gray-500">No quality history recorded.</p>
                ) : (
                  <div className="mt-3 overflow-x-auto rounded-lg border border-gray-800">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="bg-gray-950 text-left text-xs font-medium uppercase tracking-wide text-gray-500">
                          <th className="px-4 py-2">Timestamp</th>
                          {historyColumns.map((column) => (
                            <th key={column} className="px-4 py-2">{column}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {historyEntries.map((entry, index) => (
                          <tr key={entry.timestamp ?? index} className="border-t border-gray-800 text-gray-300">
                            <td className="px-4 py-2">
                              {entry.timestamp ? new Date(entry.timestamp).toLocaleString() : 'unknown'}
                            </td>
                            {historyColumns.map((column) => {
                              const value = entry[column]
                              return (
                                <td key={column} className="px-4 py-2">
                                  {typeof value === 'number' ? formatNumber(value) : 'n/a'}
                                </td>
                              )
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

              <div>
                <h3 className="text-sm font-semibold text-white">Adapter comparison</h3>
                <div className="mt-3">
                  <QualityComparisonPanel profileId={selectedProfileId} />
                </div>
              </div>
            </section>
          )}
        </>
      )}

      {jobId && (
        <section className="rounded-xl border border-gray-800 bg-gray-900/80 p-6 shadow-lg">
          <h2 className="text-lg font-semibold text-white">Job metrics</h2>
          <p className="mt-1 text-sm text-gray-400">Conversion quality metrics for job {jobId}.</p>
          <div className="mt-4">
            <QualityMetricsPanel jobId={jobId} />
          </div>
        </section>
      )}

      <AnalysisToolsCard />
    </div>
  )
}

function AnalysisToolsCard() {
  const toast = useToastContext()

  const profilesQuery = useQuery({
    queryKey: ['profiles'],
    queryFn: () => apiService.listProfiles(),
  })
  const conversionOptionsQuery = useQuery({
    queryKey: ['qualityConversionOptions'],
    queryFn: () => apiService.listQualityConversionOptions(),
    refetchInterval: 15000,
  })

  const profileOptions = profilesQuery.data ?? []
  const allConversions = conversionOptionsQuery.data?.conversions ?? []
  const comparisonSources = (conversionOptionsQuery.data?.sources ?? []).filter(
    (source) => source.conversions.length >= 2
  )
  const conversionOptionsState: 'loading' | 'error' | 'ready-empty' | 'ready-with-options' =
    conversionOptionsQuery.isLoading
      ? 'loading'
      : conversionOptionsQuery.error
        ? 'error'
        : allConversions.length === 0
          ? 'ready-empty'
          : 'ready-with-options'
  const analyzeOptionLabel =
    conversionOptionsState === 'loading'
      ? 'Loading quality-ready artifacts...'
      : conversionOptionsState === 'error'
        ? 'Quality options unavailable'
        : conversionOptionsState === 'ready-empty'
          ? 'No quality-ready conversion artifacts'
          : 'Choose a quality-ready conversion'
  const compareSourceLabel =
    conversionOptionsState === 'loading'
      ? 'Loading quality-ready sources...'
      : conversionOptionsState === 'error'
        ? 'Quality options unavailable'
        : comparisonSources.length === 0
          ? 'Need two quality-ready artifacts for one source'
          : 'Choose a source'
  const [analyzeConversionId, setAnalyzeConversionId] = useState('')
  const [analyzeProfileId, setAnalyzeProfileId] = useState('')
  const [analyzeResult, setAnalyzeResult] = useState<ConversionAnalysis | null>(null)
  const [analyzing, setAnalyzing] = useState(false)

  const [compareSourceId, setCompareSourceId] = useState('')
  const [compareProfileId, setCompareProfileId] = useState('')
  const [selectedCompareIds, setSelectedCompareIds] = useState<string[]>([])
  const [compareResult, setCompareResult] = useState<MethodologyComparison | null>(null)
  const [comparing, setComparing] = useState(false)

  const selectedAnalyzeConversion =
    allConversions.find((conversion) => conversion.id === analyzeConversionId) ?? null
  const selectedCompareSource =
    comparisonSources.find((source) => source.id === compareSourceId) ?? null
  const selectedCompareIdsInSource = selectedCompareSource
    ? selectedCompareIds.filter((id) =>
        selectedCompareSource.conversions.some((conversion) => conversion.id === id)
      )
    : []
  const effectiveCompareIds = selectedCompareIdsInSource

  const modelSummaries = Array.from(
    allConversions.reduce((summary, conversion) => {
      const key =
        conversion.adapter_type ??
        conversion.active_model_type ??
        conversion.resolved_pipeline ??
        conversion.pipeline_type ??
        'Unknown model'
      const current = summary.get(key) ?? {
        label: key,
        count: 0,
        qualityTotal: 0,
        qualityCount: 0,
        rtfTotal: 0,
        rtfCount: 0,
      }
      current.count += 1
      if (typeof conversion.quality_score === 'number') {
        current.qualityTotal += conversion.quality_score
        current.qualityCount += 1
      }
      if (typeof conversion.rtf === 'number') {
        current.rtfTotal += conversion.rtf
        current.rtfCount += 1
      }
      summary.set(key, current)
      return summary
    }, new Map<string, { label: string; count: number; qualityTotal: number; qualityCount: number; rtfTotal: number; rtfCount: number }>())
      .values()
  )

  const profileSummaries = Array.from(
    allConversions.reduce((summary, conversion) => {
      const key = conversion.profile_id ?? 'unassigned'
      const current = summary.get(key) ?? {
        label: conversion.profile_name ?? conversion.profile_id ?? 'Unassigned',
        count: 0,
        qualityTotal: 0,
        qualityCount: 0,
      }
      current.count += 1
      if (typeof conversion.quality_score === 'number') {
        current.qualityTotal += conversion.quality_score
        current.qualityCount += 1
      }
      summary.set(key, current)
      return summary
    }, new Map<string, { label: string; count: number; qualityTotal: number; qualityCount: number }>())
      .values()
  )

  const handleAnalyze = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!selectedAnalyzeConversion) {
      toast.error('Select a saved conversion to analyze')
      return
    }
    setAnalyzing(true)
    try {
      const result = await apiService.analyzeConversionRecord({
        conversion_id: selectedAnalyzeConversion.id,
        target_profile_id: analyzeProfileId || selectedAnalyzeConversion.profile_id || undefined,
      })
      setAnalyzeResult(result)
      toast.success('Conversion analysis complete')
    } catch (analyzeError) {
      toast.error(analyzeError instanceof Error ? analyzeError.message : 'Conversion analysis failed')
    } finally {
      setAnalyzing(false)
    }
  }

  const handleCompare = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!selectedCompareSource || effectiveCompareIds.length < 2) {
      toast.error('Select one source and at least two completed conversions')
      return
    }
    setComparing(true)
    try {
      const result = await apiService.compareConversionRecords({
        source_id: selectedCompareSource.id,
        conversion_ids: effectiveCompareIds,
        target_profile_id: compareProfileId || undefined,
      })
      setCompareResult(result)
      toast.success('Methodology comparison complete')
    } catch (compareError) {
      toast.error(compareError instanceof Error ? compareError.message : 'Methodology comparison failed')
    } finally {
      setComparing(false)
    }
  }

  const rankingEntries: Array<[string, number | null]> = compareResult
    ? Array.isArray(compareResult.rankings)
      ? compareResult.rankings.map((entry, index) =>
          Array.isArray(entry) ? [String(entry[0]), Number(entry[1])] : [String(entry), index + 1]
        )
      : Object.entries(compareResult.rankings).map(([methodology, score]) => [methodology, score])
    : []

  const selectedConversionMeta: Array<[string, string]> = selectedAnalyzeConversion
    ? [
        ['Source', selectedAnalyzeConversion.source_label],
        ['Method', selectedAnalyzeConversion.methodology],
        ['Profile', selectedAnalyzeConversion.profile_name ?? selectedAnalyzeConversion.profile_id ?? 'Unknown'],
        [
          'Runtime',
          [
            selectedAnalyzeConversion.runtime_backend,
            selectedAnalyzeConversion.rtf != null ? `RTF ${formatNumber(selectedAnalyzeConversion.rtf)}` : null,
          ]
            .filter(Boolean)
            .join(' · ') || 'Not recorded',
        ],
      ]
    : []

  const inputClass =
    'w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-white placeholder:text-gray-500'

  return (
    <section className="rounded-xl border border-gray-800 bg-gray-900/80 p-6 shadow-lg">
      <h2 className="text-lg font-semibold text-white">Analysis tools</h2>
      <p className="mt-1 text-sm text-gray-400">
        Select saved conversion records. The backend resolves source and output artifacts; no server paths are exposed.
      </p>

      <div className="mt-5 grid grid-cols-1 gap-4 xl:grid-cols-2">
        <div className="rounded-lg border border-gray-800 bg-gray-950/50 p-4" data-testid="quality-model-summary">
          <h3 className="text-sm font-semibold text-white">Model outputs</h3>
          <div className="mt-3 space-y-2 text-sm">
            {modelSummaries.length === 0 && <div className="text-gray-500">{conversionOptionsState === 'error' ? 'Quality options unavailable.' : 'No quality-ready conversion artifacts yet.'}</div>}
            {modelSummaries.map((summary) => (
              <div key={summary.label} className="flex items-center justify-between gap-4 rounded-lg bg-gray-900 px-3 py-2">
                <span className="text-gray-200">{summary.label}</span>
                <span className="text-right text-gray-400">
                  {summary.count} output{summary.count === 1 ? '' : 's'}
                  {summary.qualityCount > 0 ? ` · avg ${formatNumber(summary.qualityTotal / summary.qualityCount)}` : ''}
                  {summary.rtfCount > 0 ? ` · RTF ${formatNumber(summary.rtfTotal / summary.rtfCount)}` : ''}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-lg border border-gray-800 bg-gray-950/50 p-4" data-testid="quality-profile-summary">
          <h3 className="text-sm font-semibold text-white">Profile outputs</h3>
          <div className="mt-3 space-y-2 text-sm">
            {profileSummaries.length === 0 && <div className="text-gray-500">{conversionOptionsState === 'error' ? 'Quality options unavailable.' : 'No profile-linked quality artifacts yet.'}</div>}
            {profileSummaries.map((summary) => (
              <div key={summary.label} className="flex items-center justify-between gap-4 rounded-lg bg-gray-900 px-3 py-2">
                <span className="text-gray-200">{summary.label}</span>
                <span className="text-right text-gray-400">
                  {summary.count} output{summary.count === 1 ? '' : 's'}
                  {summary.qualityCount > 0 ? ` · avg ${formatNumber(summary.qualityTotal / summary.qualityCount)}` : ''}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {conversionOptionsQuery.error && (
        <StatusBanner
          tone="danger"
          title="Quality options API unavailable"
          message={(conversionOptionsQuery.error as Error).message}
          testId="quality-conversion-options-error"
        />
      )}

      <div className="mt-6 grid grid-cols-1 gap-6 xl:grid-cols-2">
        <form onSubmit={handleAnalyze} className="space-y-3" data-testid="quality-analyze-form">
          <h3 className="text-sm font-semibold text-white">Analyze conversion</h3>
          <div>
            <label className="mb-1 block text-sm text-gray-400" htmlFor="analyze-conversion">Saved conversion</label>
            <select
              id="analyze-conversion"
              data-testid="quality-conversion-selector"
              value={selectedAnalyzeConversion?.id ?? ''}
              onChange={(event) => {
                setAnalyzeConversionId(event.target.value)
                setAnalyzeResult(null)
              }}
              disabled={conversionOptionsState !== 'ready-with-options'}
              className={inputClass}
            >
              <option value="">{analyzeOptionLabel}</option>
              {allConversions.map((conversion) => (
                <option key={conversion.id} value={conversion.id}>
                  {conversion.label}
                </option>
              ))}
            </select>
          </div>
          {selectedConversionMeta.length > 0 && (
            <div className="rounded-lg border border-gray-800 bg-gray-950/70 p-3 text-xs text-gray-300">
              {selectedConversionMeta.map(([label, value]) => (
                <div key={label} className="flex justify-between gap-4">
                  <span className="text-gray-500">{label}</span>
                  <span className="text-right">{value}</span>
                </div>
              ))}
            </div>
          )}
          <div>
            <label className="mb-1 block text-sm text-gray-400" htmlFor="analyze-profile">Target profile override (optional)</label>
            <select
              id="analyze-profile"
              value={analyzeProfileId}
              onChange={(event) => setAnalyzeProfileId(event.target.value)}
              className={inputClass}
            >
              <option value="">Use conversion profile</option>
              {profileOptions.map((profile) => (
                <option key={profile.profile_id} value={profile.profile_id}>
                  {profile.name ?? profile.profile_id}
                </option>
              ))}
            </select>
          </div>
          <button
            type="submit"
            disabled={analyzing || !selectedAnalyzeConversion}
            className="rounded-lg bg-cyan-600 px-4 py-2 text-sm font-medium text-white hover:bg-cyan-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {analyzing ? 'Analyzing...' : 'Analyze conversion'}
          </button>

          {analyzeResult && (
            <div className="rounded-lg border border-gray-800 bg-gray-950/70 p-4 text-sm" data-testid="quality-analyze-result">
              <div className="flex items-center justify-between">
                <span className="font-medium text-white">
                  Quality score {formatNumber(analyzeResult.quality_score)}
                </span>
                <span
                  className={clsx(
                    'rounded-full px-2 py-1 text-xs font-medium',
                    analyzeResult.passes_thresholds
                      ? 'bg-emerald-500/15 text-emerald-200'
                      : 'bg-red-500/15 text-red-300'
                  )}
                >
                  {analyzeResult.passes_thresholds ? 'Passes thresholds' : 'Below thresholds'}
                </span>
              </div>
              {analyzeResult.conversion && (
                <div className="mt-2 text-xs text-gray-400">
                  {analyzeResult.conversion.source_label} · {analyzeResult.conversion.methodology}
                </div>
              )}
              <div className="mt-3 space-y-1 text-gray-300">
                {Object.entries(analyzeResult.metrics).map(([key, value]) => (
                  <div key={key} className="flex justify-between gap-4">
                    <span className="text-gray-500">{key}</span>
                    <span>{formatNumber(value)}</span>
                  </div>
                ))}
              </div>
              {analyzeResult.threshold_failures.length > 0 && (
                <div className="mt-3 text-amber-200">
                  Failures: {analyzeResult.threshold_failures.join(', ')}
                </div>
              )}
              {analyzeResult.recommendations.length > 0 && (
                <ul className="mt-3 list-inside list-disc text-gray-400">
                  {analyzeResult.recommendations.map((recommendation) => (
                    <li key={recommendation}>{recommendation}</li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </form>

        <form onSubmit={handleCompare} className="space-y-3" data-testid="quality-compare-form">
          <h3 className="text-sm font-semibold text-white">Compare methodologies</h3>
          <div>
            <label className="mb-1 block text-sm text-gray-400" htmlFor="compare-source">Source record</label>
            <select
              id="compare-source"
              data-testid="quality-compare-source-selector"
              value={selectedCompareSource?.id ?? ''}
              onChange={(event) => {
                setCompareSourceId(event.target.value)
                setSelectedCompareIds([])
                setCompareResult(null)
              }}
              disabled={conversionOptionsState !== 'ready-with-options' || comparisonSources.length === 0}
              className={inputClass}
            >
              <option value="">{compareSourceLabel}</option>
              {comparisonSources.map((source) => (
                <option key={source.id} value={source.id}>
                  {source.label} ({source.conversions.length} outputs)
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="mb-1 block text-sm text-gray-400" htmlFor="compare-profile">Target profile override (optional)</label>
            <select
              id="compare-profile"
              value={compareProfileId}
              onChange={(event) => setCompareProfileId(event.target.value)}
              className={inputClass}
            >
              <option value="">Use conversion profile</option>
              {profileOptions.map((profile) => (
                <option key={profile.profile_id} value={profile.profile_id}>
                  {profile.name ?? profile.profile_id}
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-2">
            <span className="block text-sm text-gray-400">Completed outputs</span>
            {!selectedCompareSource && (
              <div className="rounded-lg border border-gray-800 bg-gray-950/70 p-3 text-sm text-gray-500">
                {compareSourceLabel}.
              </div>
            )}
            {selectedCompareSource?.conversions.map((conversion) => {
              const checked = effectiveCompareIds.includes(conversion.id)
              return (
                <label
                  key={conversion.id}
                  className="flex items-start gap-3 rounded-lg border border-gray-800 bg-gray-950/70 p-3 text-sm"
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => {
                      const next = checked
                        ? effectiveCompareIds.filter((id) => id !== conversion.id)
                        : [...effectiveCompareIds, conversion.id]
                      setSelectedCompareIds(next)
                      setCompareResult(null)
                    }}
                    className="mt-1"
                  />
                  <span className="min-w-0">
                    <span className="block font-medium text-gray-100">{conversion.methodology}</span>
                    <span className="block text-xs text-gray-500">
                      {conversion.profile_name ?? conversion.profile_id ?? 'Unknown profile'}
                      {conversion.rtf != null ? ` · RTF ${formatNumber(conversion.rtf)}` : ''}
                      {conversion.quality_score != null ? ` · quality ${formatNumber(conversion.quality_score)}` : ''}
                    </span>
                  </span>
                </label>
              )
            })}
          </div>
          <button
            type="submit"
            disabled={comparing || !selectedCompareSource || effectiveCompareIds.length < 2}
            className="rounded-lg bg-cyan-600 px-4 py-2 text-sm font-medium text-white hover:bg-cyan-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {comparing ? 'Comparing...' : 'Compare methodologies'}
          </button>

          {compareResult && (
            <div className="rounded-lg border border-gray-800 bg-gray-950/70 p-4 text-sm" data-testid="quality-compare-result">
              <div className="font-medium text-white">
                Best methodology: <span className="text-emerald-300">{compareResult.best_methodology}</span>
              </div>
              <div className="mt-3 overflow-hidden rounded-lg border border-gray-800">
                <div className="grid grid-cols-2 bg-gray-950 px-3 py-2 text-xs font-medium uppercase tracking-wide text-gray-500">
                  <span>Methodology</span>
                  <span>Score</span>
                </div>
                {rankingEntries.map(([methodology, score]) => (
                  <div key={methodology} className="grid grid-cols-2 border-t border-gray-800 px-3 py-2 text-gray-300">
                    <span className={methodology === compareResult.best_methodology ? 'text-emerald-300' : undefined}>
                      {methodology}
                    </span>
                    <span>{score == null ? 'Ranked' : formatNumber(score)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </form>
      </div>
    </section>
  )
}

function StatCard({
  label,
  value,
  tone = 'default',
}: {
  label: string
  value: string | number
  tone?: 'default' | 'warning' | 'danger'
}) {
  return (
    <div
      className={clsx(
        'rounded-xl border p-4 shadow-lg',
        tone === 'danger'
          ? 'border-red-500/20 bg-red-500/10'
          : tone === 'warning'
            ? 'border-amber-500/20 bg-amber-500/10'
            : 'border-gray-800 bg-gray-900/80'
      )}
    >
      <div className="text-sm text-gray-300">{label}</div>
      <div className="mt-2 text-2xl font-semibold text-white">{value}</div>
    </div>
  )
}
