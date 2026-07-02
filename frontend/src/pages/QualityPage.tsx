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
  const profileOptions = profilesQuery.data ?? []

  const [analyzeForm, setAnalyzeForm] = useState({
    source: '',
    converted: '',
    profileId: '',
    methodology: '',
  })
  const [analyzeResult, setAnalyzeResult] = useState<ConversionAnalysis | null>(null)
  const [analyzing, setAnalyzing] = useState(false)

  const [compareSource, setCompareSource] = useState('')
  const [compareProfileId, setCompareProfileId] = useState('')
  const [compareRows, setCompareRows] = useState([
    { methodology: '', path: '' },
    { methodology: '', path: '' },
  ])
  const [compareResult, setCompareResult] = useState<MethodologyComparison | null>(null)
  const [comparing, setComparing] = useState(false)

  const handleAnalyze = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!analyzeForm.source.trim() || !analyzeForm.converted.trim()) {
      toast.error('Source and converted audio paths are required')
      return
    }
    setAnalyzing(true)
    try {
      const result = await apiService.analyzeConversion({
        source_audio: analyzeForm.source.trim(),
        converted_audio: analyzeForm.converted.trim(),
        target_profile_id: analyzeForm.profileId || undefined,
        methodology: analyzeForm.methodology.trim() || undefined,
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
    const outputs: Record<string, string> = {}
    for (const row of compareRows) {
      if (row.methodology.trim() && row.path.trim()) {
        outputs[row.methodology.trim()] = row.path.trim()
      }
    }
    if (!compareSource.trim() || Object.keys(outputs).length === 0) {
      toast.error('Source path and at least one methodology/path pair are required')
      return
    }
    setComparing(true)
    try {
      const result = await apiService.compareMethodologies({
        source_audio: compareSource.trim(),
        target_profile_id: compareProfileId || undefined,
        converted_outputs: outputs,
      })
      setCompareResult(result)
      toast.success('Methodology comparison complete')
    } catch (compareError) {
      toast.error(compareError instanceof Error ? compareError.message : 'Methodology comparison failed')
    } finally {
      setComparing(false)
    }
  }

  const rankingEntries: Array<[string, number]> = compareResult
    ? Array.isArray(compareResult.rankings)
      ? compareResult.rankings
      : Object.entries(compareResult.rankings)
    : []

  const inputClass =
    'w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-white placeholder:text-gray-500'

  return (
    <section className="rounded-xl border border-gray-800 bg-gray-900/80 p-6 shadow-lg">
      <h2 className="text-lg font-semibold text-white">Analysis tools</h2>
      <p className="mt-1 text-sm text-gray-400">
        Paths are file paths on the server (operator tool). Results come from the live analysis API.
      </p>

      <div className="mt-6 grid grid-cols-1 gap-6 xl:grid-cols-2">
        <form onSubmit={handleAnalyze} className="space-y-3" data-testid="quality-analyze-form">
          <h3 className="text-sm font-semibold text-white">Analyze conversion</h3>
          <div>
            <label className="mb-1 block text-sm text-gray-400" htmlFor="analyze-source">Source audio path</label>
            <input
              id="analyze-source"
              value={analyzeForm.source}
              onChange={(event) => setAnalyzeForm((prev) => ({ ...prev, source: event.target.value }))}
              placeholder="/path/on/server/source.wav"
              className={inputClass}
            />
          </div>
          <div>
            <label className="mb-1 block text-sm text-gray-400" htmlFor="analyze-converted">Converted audio path</label>
            <input
              id="analyze-converted"
              value={analyzeForm.converted}
              onChange={(event) => setAnalyzeForm((prev) => ({ ...prev, converted: event.target.value }))}
              placeholder="/path/on/server/converted.wav"
              className={inputClass}
            />
          </div>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div>
              <label className="mb-1 block text-sm text-gray-400" htmlFor="analyze-profile">Target profile (optional)</label>
              <select
                id="analyze-profile"
                value={analyzeForm.profileId}
                onChange={(event) => setAnalyzeForm((prev) => ({ ...prev, profileId: event.target.value }))}
                className={inputClass}
              >
                <option value="">None</option>
                {profileOptions.map((profile) => (
                  <option key={profile.profile_id} value={profile.profile_id}>
                    {profile.name ?? profile.profile_id}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="mb-1 block text-sm text-gray-400" htmlFor="analyze-methodology">Methodology (optional)</label>
              <input
                id="analyze-methodology"
                value={analyzeForm.methodology}
                onChange={(event) => setAnalyzeForm((prev) => ({ ...prev, methodology: event.target.value }))}
                placeholder="quality_seedvc"
                className={inputClass}
              />
            </div>
          </div>
          <button
            type="submit"
            disabled={analyzing}
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
            <label className="mb-1 block text-sm text-gray-400" htmlFor="compare-source">Source audio path</label>
            <input
              id="compare-source"
              value={compareSource}
              onChange={(event) => setCompareSource(event.target.value)}
              placeholder="/path/on/server/source.wav"
              className={inputClass}
            />
          </div>
          <div>
            <label className="mb-1 block text-sm text-gray-400" htmlFor="compare-profile">Target profile (optional)</label>
            <select
              id="compare-profile"
              value={compareProfileId}
              onChange={(event) => setCompareProfileId(event.target.value)}
              className={inputClass}
            >
              <option value="">None</option>
              {profileOptions.map((profile) => (
                <option key={profile.profile_id} value={profile.profile_id}>
                  {profile.name ?? profile.profile_id}
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-2">
            <span className="block text-sm text-gray-400">Converted outputs (methodology → server path)</span>
            {compareRows.map((row, index) => (
              <div key={index} className="flex gap-2">
                <input
                  value={row.methodology}
                  onChange={(event) => setCompareRows((prev) =>
                    prev.map((item, i) => (i === index ? { ...item, methodology: event.target.value } : item))
                  )}
                  placeholder="methodology"
                  aria-label={`Methodology ${index + 1}`}
                  className={clsx(inputClass, 'w-40 flex-none')}
                />
                <input
                  value={row.path}
                  onChange={(event) => setCompareRows((prev) =>
                    prev.map((item, i) => (i === index ? { ...item, path: event.target.value } : item))
                  )}
                  placeholder="/path/on/server/output.wav"
                  aria-label={`Converted path ${index + 1}`}
                  className={inputClass}
                />
                <button
                  type="button"
                  onClick={() => setCompareRows((prev) => prev.filter((_, i) => i !== index))}
                  disabled={compareRows.length <= 1}
                  aria-label={`Remove row ${index + 1}`}
                  className="rounded-lg border border-gray-700 bg-gray-800 px-3 text-sm text-gray-300 hover:bg-gray-700 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Remove
                </button>
              </div>
            ))}
            <button
              type="button"
              onClick={() => setCompareRows((prev) => [...prev, { methodology: '', path: '' }])}
              className="rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200 hover:bg-gray-700"
            >
              Add row
            </button>
          </div>
          <button
            type="submit"
            disabled={comparing}
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
                    <span>{formatNumber(score)}</span>
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
